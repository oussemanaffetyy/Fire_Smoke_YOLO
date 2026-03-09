FROM nodered/node-red:latest

RUN npm install --unsafe-perm --no-update-notifier --no-fund node-red-dashboard
