import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests
import serial

# =========================
# CONFIG
# =========================
PORT = "COM4"
BAUD = 115200

API_URL = "https://satpi-backend.onrender.com/telemetry"

LOCAL_JSON = Path("latest_telemetry.json")

CSV_FIELDS = [
    "lat", "lon", "alt", "vel", "temp", "press",
    "alt_press", "temps_txt", "temps", "camX", "camY", "pc_rebut_ts",
]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
HISTORY_CSV = Path(f"GroundStationBernatelFerrer_{timestamp}.csv")

SEND_TO_CLOUD    = True
CLOUD_TIMEOUT    = 3
LOOP_SLEEP       = 0.2

# Reconexió sèrie
RECONNECT_DELAY  = 3
RECONNECT_MAX    = 0       # 0 = infinit

# Reintents POST al cloud
CLOUD_RETRIES    = 2
CLOUD_RETRY_WAIT = 0.5

# Posició per defecte quan no hi ha fix GPS (aeroport d'Alguaire, Lleida)
GPS_DEFAULT_LAT = 41.72857426231821
GPS_DEFAULT_LON =  0.5434698388285842
GPS_DEFAULT_ALT = 229.0

# =========================
# LOGGING
# =========================
LOG_FILE = Path(f"bridge_log_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("bridge")


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
# Format esperat (8 o 10 camps):
#   lat,lon,alt,vel,temp,press,alt_press,temps[,camX,camY]
# Exemple complet:
#   41.564421,2.006014,508.4,8.23,19.8,999.1,507.2,00:00:06,right,center
# Exemple sense GPS:
#   None,None,None,None,42.22,979.24,287.14,10:40:23
# =========================
def _pf(valor: str):
    """Parse float tolerant: retorna None si es "None", buit o no numeric."""
    if valor is None:
        return None
    v = valor.strip()
    if v.lower() in ("none", "", "null", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_line(line: str):
    parts = [p.strip() for p in line.strip().split(",")]

    if len(parts) not in (8, 10):
        return None

    try:
        temps_txt = parts[7]
        temps_seg = hhmmss_a_segons(temps_txt)
        if temps_seg is None:
            return None

        lat       = _pf(parts[0])
        lon       = _pf(parts[1])
        alt       = _pf(parts[2])
        vel       = _pf(parts[3])
        temp      = _pf(parts[4])
        press     = _pf(parts[5])
        alt_press = _pf(parts[6])

        # Si no hi ha cap sensor bàsic, no te sentit enviar
        if temp is None and press is None and alt_press is None:
            return None

        return {
            "lat":       lat       if lat       is not None else GPS_DEFAULT_LAT,
            "lon":       lon       if lon       is not None else GPS_DEFAULT_LON,
            "alt":       alt       if alt       is not None else GPS_DEFAULT_ALT,
            "vel":       vel       if vel       is not None else 0.0,
            "temp":      temp      if temp      is not None else 0.0,
            "press":     press     if press     is not None else 0.0,
            "alt_press": alt_press if alt_press is not None else 0.0,
            "temps_txt": temps_txt,
            "temps":     float(temps_seg),
            "camX":      str(parts[8]) if len(parts) == 10 else "center",
            "camY":      str(parts[9]) if len(parts) == 10 else "center",
            "_gps_real": lat is not None and lon is not None,
        }

    except Exception:
        return None


# =========================
# LECTURA ANTI-LAG
# =========================
def llegir_ultima_linia(ser):
    """
    Buida el buffer serie i retorna nomes l'ultima linia completa.
    Evita processar dades antigues acumulades.
    Llanca SerialException si el port es desconnecta.
    """
    last_line = None
    try:
        while ser.in_waiting > 0:
            raw  = ser.readline()
            line = raw.decode(errors="ignore").strip()
            if line:
                last_line = line
    except serial.SerialException:
        raise
    except Exception as e:
        log.warning(f"Error llegint buffer: {e}")
        return None
    return last_line


# =========================
# GUARDAT LOCAL (atomic)
# =========================
def guardar_local(data: dict):
    d = {k: v for k, v in data.items() if not k.startswith("_")}
    try:
        tmp = LOCAL_JSON.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, LOCAL_JSON)
    except Exception as e:
        log.error(f"Error guardant JSON local: {e}")


# =========================
# CSV HISTORIAL
# =========================
def inicialitzar_csv():
    if not HISTORY_CSV.exists():
        try:
            with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
            log.info(f"CSV creat: {HISTORY_CSV}")
        except Exception as e:
            log.error(f"No s'ha pogut crear el CSV: {e}")


def afegir_a_historial_csv(data: dict):
    d = {k: v for k, v in data.items() if not k.startswith("_")}
    try:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(d)
    except Exception as e:
        log.error(f"Error escrivint CSV: {e}")


# =========================
# CLOUD (amb reintents)
# =========================
def enviar_cloud(session: requests.Session, data: dict):
    if not SEND_TO_CLOUD:
        return

    d = {k: v for k, v in data.items() if not k.startswith("_")}

    for intent in range(1 + CLOUD_RETRIES):
        try:
            r = session.post(API_URL, json=d, timeout=CLOUD_TIMEOUT)

            if r.status_code == 200:
                gps_tag = "" if data.get("_gps_real") else " [GPS=ALGUAIRE]"
                log.info(
                    f"OK  hora={data['temps_txt']}  "
                    f"lat={data['lat']:.5f} lon={data['lon']:.5f}  "
                    f"alt={data['alt']:.1f}m  temp={data['temp']:.1f}C"
                    f"{gps_tag}"
                )
                return

            else:
                log.warning(f"POST {r.status_code} (intent {intent+1}): {r.text[:160]}")

        except requests.exceptions.Timeout:
            log.warning(f"Cloud timeout (intent {intent+1}/{1+CLOUD_RETRIES})")
        except requests.exceptions.ConnectionError:
            log.warning(f"Cloud sense connexio (intent {intent+1}/{1+CLOUD_RETRIES})")
        except requests.exceptions.RequestException as e:
            log.warning(f"Cloud error: {e} (intent {intent+1})")

        if intent < CLOUD_RETRIES:
            time.sleep(CLOUD_RETRY_WAIT)

    log.error("No s'ha pogut enviar al cloud despres de tots els intents.")


# =========================
# CONNEXIO SERIE (amb reconexio)
# =========================
def obrir_port() -> serial.Serial:
    """Intenta obrir el port serie indefinidament fins aconseguir-ho."""
    intents = 0
    while True:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=0.1)
            log.info(f"Port {PORT} obert correctament.")
            return ser
        except serial.SerialException as e:
            intents += 1
            log.warning(f"No s'ha pogut obrir {PORT} (intent {intents}): {e}")
            if RECONNECT_MAX and intents >= RECONNECT_MAX:
                log.error("Maxim d'intents de connexio assolit. Sortint.")
                sys.exit(1)
            log.info(f"Reintentant en {RECONNECT_DELAY} s...")
            time.sleep(RECONNECT_DELAY)


# =========================
# MAIN LOOP
# =========================
def main():
    log.info("=" * 55)
    log.info("  BRIDGE SATPI26 -- arrencant")
    log.info(f"  Port: {PORT}  Baud: {BAUD}")
    log.info(f"  Cloud: {API_URL}")
    log.info(f"  Log:   {LOG_FILE}")
    log.info("=" * 55)

    session = requests.Session()
    inicialitzar_csv()

    ultima_hora_processada = None
    linies_invalides_seq   = 0

    while True:   # bucle extern: reconexio automatica
        ser = obrir_port()
        time.sleep(2)
        ser.reset_input_buffer()
        log.info("Bridge actiu. Llegint l'ultima dada disponible.\n")

        try:
            while True:   # bucle intern: lectura normal
                try:
                    line = llegir_ultima_linia(ser)
                except serial.SerialException as e:
                    log.error(f"Port serie desconnectat: {e}. Reconectant...")
                    break  # surt al bucle extern -> reconexio

                if not line:
                    time.sleep(LOOP_SLEEP)
                    continue

                data = parse_line(line)

                if data is None:
                    linies_invalides_seq += 1
                    if linies_invalides_seq <= 3 or linies_invalides_seq % 20 == 0:
                        log.debug(f"Linia ignorada ({linies_invalides_seq}): {line[:80]}")
                    time.sleep(LOOP_SLEEP)
                    continue

                linies_invalides_seq = 0

                if data["temps_txt"] == ultima_hora_processada:
                    time.sleep(LOOP_SLEEP)
                    continue

                ultima_hora_processada = data["temps_txt"]
                data["pc_rebut_ts"]    = time.time()

                guardar_local(data)
                afegir_a_historial_csv(data)
                enviar_cloud(session, data)

                time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            log.info("Aturat per l'usuari (Ctrl+C).")
            break

        except Exception as e:
            log.error(f"Error inesperat al bucle principal: {e}")
            log.debug(traceback.format_exc())
            log.info(f"Esperant {RECONNECT_DELAY} s abans de reconectar...")
            time.sleep(RECONNECT_DELAY)

        finally:
            try:
                if ser and ser.is_open:
                    ser.close()
                    log.info("Port serie tancat.")
            except Exception:
                pass

    session.close()
    log.info("Bridge tancat correctament.")


if __name__ == "__main__":
    main()
