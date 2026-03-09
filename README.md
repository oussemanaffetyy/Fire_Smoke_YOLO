# Fire_Smoke_YOLO

Projet Python prêt à l'emploi pour détection **Smoke/Fire** en temps réel via webcam.

Le projet inclut déjà les modèles extraits depuis vos ZIP:
- `models/best.pt`
- `models/best.tflite`
- `models/data.yaml`
- `models/classes.txt`

Architecture recommandée:

`ESP32-CAM stream -> Python YOLO -> MQTT -> Node-RED dashboard / alerts`

## 1) Pré-requis

- Python **3.10 ou 3.11** (3.11 recommandé)
- Webcam locale
- Windows / macOS / Linux

GPU n'est pas obligatoire. Le script fonctionne en CPU.

Note importante:
- Python 3.13 n'est pas recommandé ici (compatibilité des dépendances).

## 2) Installation rapide

### macOS / Linux

```bash
cd Fire_Smoke_YOLO
bash scripts/setup.sh
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
cd Fire_Smoke_YOLO
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

## 3) Lancer la caméra

```bash
python src/run_camera.py
```

Touches:
- `q` pour quitter.

## 4) Commandes utiles

### Forcer CPU

```bash
python src/run_camera.py --device cpu
```

### PC faible (plus stable)

```bash
python src/run_camera.py --device cpu --imgsz 416 --width 640 --height 480 --skip-frames 2
```

### PC fort / GPU

```bash
python src/run_camera.py --device auto --imgsz 960 --width 1280 --height 720
```

### Changer caméra (si 2 caméras)

```bash
python src/run_camera.py --source 1
```

### Stream ESP32-CAM (HTTP MJPEG)

```bash
python src/run_camera.py --source "http://192.168.1.23:81/stream" --model models/best.pt --device cpu --imgsz 320 --skip-frames 4 --conf 0.30
```

Le script reconnecte automatiquement si le flux coupe.

### Stream ESP32-CAM + MQTT pour Node-RED

```bash
python src/run_camera.py --source "http://192.168.1.23:81/stream" --model models/best.pt --device cpu --imgsz 320 --skip-frames 4 --conf 0.30 --mqtt-host 127.0.0.1 --mqtt-topic-prefix factory/fire_smoke --mqtt-source-id esp32cam_1
```

### Enregistrer la vidéo annotée

```bash
python src/run_camera.py --save outputs/detect.mp4
```

## 5) Détails du script

Fichier principal: `src/run_camera.py`

Fonctions clés:
- Sélection auto du device (`cuda`/`mps`/`cpu`)
- Inférence temps réel YOLO
- Affichage des boxes + panneau status (`ALERT: FIRE/SMOKE`)
- Paramètres réglables via CLI
- Publication MQTT optionnelle (`/raw`, `/alert`, `/status`)

## 6) Node-RED + MQTT

Fichiers fournis:
- `deploy/docker-compose.yml`
- `deploy/mosquitto.conf`
- `node_red/fire_smoke_dashboard_flow.json`

Lancement rapide:

```bash
cd deploy
docker compose up -d
```

Puis:
1. Ouvrir `http://localhost:1880`
2. Importer `node_red/fire_smoke_dashboard_flow.json`
3. Ouvrir `http://localhost:1880/ui`

Le conteneur Node-RED installe `node-red-dashboard` automatiquement au build.

Topics utilisés:
- `factory/fire_smoke/raw`
- `factory/fire_smoke/alert`
- `factory/fire_smoke/status`

## 7) Déplacer le projet vers un autre PC

Copiez tout le dossier `Fire_Smoke_YOLO`, puis:
1. Créer/activer `.venv`
2. `pip install -r requirements.txt`
3. `python src/run_camera.py`

Les modèles sont déjà dans `models/`, donc pas besoin de re-extraire les ZIP.

## 8) Labels du modèle

Selon vos fichiers:
- `0 -> Smoke`
- `1 -> Fire`
