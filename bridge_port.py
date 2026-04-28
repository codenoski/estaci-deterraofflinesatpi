import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests
import serial
import os

# =========================
# CONFIG
# =========================
PORT = "COM3"
BAUD = 115200

API_URL = "https://satpi-backend.onrender.com/telemetry"

LOCAL_JSON = Path("latest_telemetry.json")

CSV_FIELDS = [
    "lat",
    "lon",
    "alt",
    "vel",
    "temp",
    "press",
    "alt_press",
    "temps_txt",
    "temps",
    "camX",
    "camY",
    "pc_rebut_ts",
]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
HISTORY_CSV = Path(f"GroundStationBernatelFerrer_{timestamp}.csv")

SEND_TO_CLOUD = True
CLOUD_TIMEOUT = 3
LOOP_SLEEP = 0.2


# =========================
# TEMPS
# =========================
def hhmmss_a_segons(text):
    try:
        h, m, s = text.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


# =========================
# PARSER
# format esperat:
# lat,lon,alt,vel,temp,press,alt_press,temps,camX,camY
# exemple:
# 41.564421,2.006014,508.4,8.23,19.8,999.1,507.2,00:00:06,right,center
# =========================
def parse_line(line: str):
    parts = [p.strip() for p in line.strip().split(",")]

    # Compat: 8 camps (sense càmera) o 10 camps (amb càmera)
    if len(parts) not in (8, 10):
        return None

    try:
        temps_txt = parts[7]
        temps_segons = hhmmss_a_segons(temps_txt)

        if temps_segons is None:
            return None

        return {
            "lat": float(parts[0]),
            "lon": float(parts[1]),
            "alt": float(parts[2]),
            "vel": float(parts[3]),
            "temp": float(parts[4]),
            "press": float(parts[5]),
            "alt_press": float(parts[6]),
            "temps_txt": temps_txt,
            "temps": float(temps_segons),
            # camX/camY arriben com a string ("left"/"right"/"center")
            "camX": str(parts[8]) if len(parts) == 10 else "center",
            "camY": str(parts[9]) if len(parts) == 10 else "center",
        }

    except Exception:
        return None


# =========================
# LECTURA ANTI-LAG
# =========================
def llegir_ultima_linia(ser):
    """
    Buida el buffer sèrie i retorna només l'última línia completa.
    Això evita que el bridge vagi processant dades antigues acumulades.
    """
    last_line = None

    try:
        while ser.in_waiting > 0:
            raw = ser.readline()
            line = raw.decode(errors="ignore").strip()

            if line:
                last_line = line

    except Exception:
        return None

    return last_line


# =========================
# GUARDAT LOCAL
# =========================
def guardar_local(data: dict):
    try:
        tmp_path = LOCAL_JSON.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, LOCAL_JSON)
    except Exception as e:
        print("Error guardant JSON local:", e)


def inicialitzar_csv():
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def afegir_a_historial_csv(data: dict):
    try:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow(data)
    except Exception as e:
        print("Error escrivint historial CSV:", e)


# =========================
# CLOUD
# =========================
def enviar_cloud(session, data):
    if not SEND_TO_CLOUD:
        return

    try:
        r = session.post(API_URL, json=data, timeout=CLOUD_TIMEOUT)

        if r.status_code == 200:
            print(
                f"OK cloud/local -> hora={data['temps_txt']} "
                f"alt={data['alt']:.1f} vel={data['vel']:.2f}"
            )
        else:
            print(f"POST ERROR {r.status_code}: {r.text[:120]}")

    except requests.exceptions.RequestException as e:
        print("Error xarxa cloud:", e)


# =========================
# MAIN
# =========================
def main():
    ser = None
    session = requests.Session()

    try:
        print(f"Connectant a {PORT}...")
        ser = serial.Serial(PORT, BAUD, timeout=0.1)

        time.sleep(2)

        # Neteja inicial del buffer vell
        ser.reset_input_buffer()

        inicialitzar_csv()

        print(f"Historial CSV: {HISTORY_CSV}")
        print("Bridge PRO actiu. Llegint només l'última dada disponible.\n")

        ultima_hora_processada = None

        while True:
            line = llegir_ultima_linia(ser)

            if not line:
                time.sleep(LOOP_SLEEP)
                continue

            data = parse_line(line)

            if data is None:
                print("Línia no vàlida:", line)
                time.sleep(LOOP_SLEEP)
                continue

            # Evita duplicats exactes
            if data["temps_txt"] == ultima_hora_processada:
                time.sleep(LOOP_SLEEP)
                continue

            ultima_hora_processada = data["temps_txt"]

            # Timestamp PC real de recepció (per calcular retard real a la GS)
            data["pc_rebut_ts"] = time.time()

            # 1. Actualització local immediata
            guardar_local(data)

            # 2. Historial complet de dades processades
            afegir_a_historial_csv(data)

            # 3. Enviament cloud
            enviar_cloud(session, data)

            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        print("\nAturat per l'usuari.")

    except serial.SerialException as e:
        print(f"Error port sèrie: {e}")

    except Exception as e:
        print(f"Error inesperat: {e}")

    finally:
        if ser and ser.is_open:
            ser.close()
        session.close()
        print("Pont tancat correctament.")


if __name__ == "__main__":
    main()
