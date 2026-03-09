# Node-RED Integration

Architecture recommandee:

`ESP32-CAM stream -> run_camera.py -> MQTT -> Node-RED dashboard / alerts`

Topics publishes par `run_camera.py`:
- `factory/fire_smoke/raw`
- `factory/fire_smoke/alert`
- `factory/fire_smoke/status`

Fichier a importer dans Node-RED:
- `node_red/fire_smoke_dashboard_flow.json`

Modules Node-RED a installer:
- `node-red-dashboard`

Dashboard:
- `http://localhost:1880/ui`

Le flow suppose un broker MQTT local:
- host: `127.0.0.1`
- port: `1883`
