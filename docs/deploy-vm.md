# Live deploy on a single VM (Docker Compose)

Runs the whole stack on one cloud VM (e.g. an Oracle Always-Free **Ampere A1**, ≥4 GB) and
exposes Grafana on the VM's public IP — the live link. Same Phase-1 compose stack, hardened for
public exposure (`docker-compose.public.yml`: real Grafana password, no anonymous access).

> The A1 shape is arm64 — building on the VM (`--build`) produces an arm image natively, no
> cross-build needed. The 1 GB x86 micro is too small; use the A1.

## 1. On the VM — install Docker
```bash
ssh -i <your-key> ubuntu@<vm-public-ip>      # opc@ for Oracle Linux
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
```

## 2. Clone + bring it up
```bash
git clone https://github.com/nanu1605/mlops-drift-retrain.git
cd mlops-drift-retrain/deploy/docker
export GRAFANA_ADMIN_PASSWORD='<a-strong-password>'
docker compose -f docker-compose.yml -f docker-compose.public.yml up -d --build
```
`seed` trains the champion and exits; serving + controller + Prometheus + Grafana stay up
(`restart: unless-stopped`).

## 3. Open the firewall — BOTH layers (the Oracle gotcha)
Only port **3000** (Grafana). Everything else stays internal.

**a. Cloud — OCI security list / NSG:** Console → VCN → the instance's subnet → Security List →
add ingress: source `0.0.0.0/0`, TCP, dest port `3000`.

**b. OS firewall:** Oracle images ship iptables that drop everything but SSH. Open 3000 and
persist:
```bash
sudo iptables -I INPUT 6 -p tcp --dport 3000 -j ACCEPT
sudo netfilter-persistent save        # Ubuntu; Oracle Linux: firewall-cmd --add-port=3000/tcp --permanent && firewall-cmd --reload
```
Miss either layer and the page just hangs.

## 4. The live link
`http://<vm-public-ip>:3000` → log in `admin` / your password → **Drift & Retrain** dashboard.

Drive the recovery so the dashboard tells the story:
```bash
docker compose exec controller make replay
```
`realized_f1` dips ~0.28 → recovers ~0.83, serving `model_version` 1 → 2. Share that panel.

## 5. Notes
- Health check: `curl localhost:8000/health` on the VM (serving is loopback-only — not public).
- Rebuild after a code change: `docker compose ... up -d --build` then `down -v` first if you want
  fresh state (the shared volume is seeded from the image on first run).
- Stop the demo: `docker compose -f docker-compose.yml -f docker-compose.public.yml down`
  (add `-v` to wipe state).
- For a domain + TLS, front Grafana with Caddy/nginx on 443 and close 3000.
