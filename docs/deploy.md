# Deployment runbook

Runs the full **drift → retrain → promote → reload** loop in containers — first in Docker
Compose, then on minikube with Prometheus + Grafana via kube-prometheus-stack.

One image, four entrypoints (the command selects the workload):

| Workload   | Command                  | Notes                                  |
|------------|--------------------------|----------------------------------------|
| seed/train | `make train`             | registers the first `@champion`        |
| serve      | `make up HOST=0.0.0.0`   | uvicorn on `:8000`                      |
| controller | `make loop`              | drift loop + monitor metrics on `:9100`|
| monitor    | `make monitor`           | one-shot drift report (optional)       |

State (`mlflow.db`, `mlartifacts/`, `data/serving/*.db`, `metrics/`) is single-writer per
SQLite file and lives on one shared volume — serving writes the request DB, the controller
writes the registry. Never run a second writer against either.

---

## Phase 1 — Docker Compose

```bash
cd deploy/docker
docker compose up --build
```

`seed` trains and exits 0; `serving` becomes healthy; `controller` starts and exposes `:9100`;
Prometheus (`:9090`) and Grafana (`:3000`, anonymous admin) come up.

Verify:
```bash
curl localhost:8000/health                      # 200
docker compose exec serving make smoke          # 200 predict + 400 bad-input + 200 metrics
curl -s localhost:9090/api/v1/targets           # jobs "serving" and "drift" both up
docker compose exec controller make replay      # stream the drift period to serving
```
After replay, the controller detects drift, retrains, promotes, and `POST /reload`s serving.
Watch `realized_f1` on `localhost:9100/metrics` jump (~0.28 → ~0.83) and the serving
`model_version` advance. The **Drift & Retrain** dashboard shows the dip → recovery.

Tear down (`-v` also wipes the state volume):
```bash
docker compose down -v
```

> Note: the shared volume is seeded from the image on first run. After rebuilding the image,
> `docker compose down -v` before `up` so the new code isn't shadowed by stale volume contents.

---

## Phase 2 — Minikube

Build into minikube's own daemon so `imagePullPolicy: IfNotPresent` finds the image:
```bash
eval $(minikube docker-env)
docker build -f deploy/docker/serving.Dockerfile -t mlops-drift:local .
```

Install monitoring (Grafana + Prometheus operator ship with the chart):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

Apply the app (namespace → PVC → seed Job → serving → controller → ServiceMonitors):
```bash
kubectl apply -f deploy/k8s/
kubectl -n mlops wait --for=condition=complete job/seed-train --timeout=300s
kubectl -n mlops get pods            # serving + controller 1/1 Running
```

Because a PVC mounted at `/app` shadows the baked code (k8s doesn't auto-populate volumes),
the seed Job's `populate-code` initContainer copies `/opt/app` into the empty PVC once;
serving and controller block on the `@champion` existing before they start.

Verify:
```bash
kubectl -n mlops exec deploy/serving -- curl -s localhost:8000/health      # 200
kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 9090:9090 &
#   /targets: serving + controller ServiceMonitors both up
kubectl -n monitoring port-forward svc/kps-grafana 3000:80 &
#   Grafana: "Intrusion Serving" + "Drift & Retrain" dashboards (admin / `prom-operator`)
kubectl -n mlops exec deploy/controller -- make replay                     # drive drift
```
Same recovery as Phase 1, now scraped through ServiceMonitors and rendered by the chart's
Grafana. `drift_detected` stays 1 while the served window is all-drift — the honest recovery
signal is the per-batch `realized_f1` series, not the promotion-gate numbers (the holdout
overlaps retrain data and is optimistic).

Tear down:
```bash
kubectl delete -f deploy/k8s/ ; kubectl delete ns mlops
helm uninstall kps -n monitoring ; kubectl delete ns monitoring
```

---

## Scale-out (not built)

Single node only: SQLite + in-process model swap + a ReadWriteOnce PVC pin everything to one
serving worker on one node. The real scale-out path is MLflow on **Postgres + S3/MinIO**, after
which serving can be `replicas: N`, retrain becomes a standalone K8s **Job**, and the PVC
constraint disappears.
