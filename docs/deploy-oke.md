# Live deploy on Oracle OKE (Always Free)

Runs the stack on a managed single-node ARM cluster and exposes Grafana on a public IP — the
live link. Free-forever on Oracle's Always-Free tier (1 OKE basic cluster, ARM Ampere A1 up to
4 OCPU / 24 GB, 1 flexible load balancer, block volumes within the free allowance).

Everything Oracle-side (account, capacity, registry creds) is yours to run — this repo ships the
tuned helm values (`deploy/helm/kps-values-oke.yaml`) and the image/PVC overlay
(`deploy/k8s/oke/`).

> Single-node by design: SQLite + in-process model swap + a ReadWriteOnce PVC. One node is
> correct here. Only Grafana is public; serving + controller stay ClusterIP-internal.

---

## 0. Prereqs (local)
- Oracle Cloud Always-Free account.
- `oci` CLI (`oci setup config`), `kubectl`, `helm`, `docker buildx`.
- A registry for the image. Docker Hub **public** repo is simplest (no pull secret). OCIR works
  too (native + free) if you want it private.

## 1. Provision the cluster
Console → **Kubernetes Clusters (OKE)** → **Create cluster** → **Quick create**:
- Cluster type **Basic** (free control plane).
- Node pool: shape **VM.Standard.A1.Flex**, **1 node**, **4 OCPU / 24 GB** (the whole free A1
  budget on one node — kube-prometheus-stack is heavy).
- Public endpoint + public workers (simplest; quick-create wires the VCN/security lists).

> **Capacity gotcha:** Always-Free A1 is often "out of host capacity" in busy regions. Retry, or
> create in a quieter home region.

Wire kubeconfig:
```bash
oci ce cluster create-kubeconfig --cluster-id <ocid> --file ~/.kube/config \
  --region <region> --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT
kubectl get nodes        # 1 Ready arm64 node
```

## 2. Get the arm64 image
A1 is arm64; the local image was x86 — you need a multi-arch image in a registry.

**Default — CI builds it.** The [`image`](../.github/workflows/image.yml) workflow pushes a
multi-arch (`amd64`+`arm64`) manifest to `ghcr.io/<owner>/mlops-drift:latest` on every push to
master (or run it via *Actions → image → Run workflow*). One-time: make the GHCR package
**public** (GitHub → Packages → `mlops-drift` → Package settings → visibility) so the node pulls
it without a secret. The overlay already points here.

**Fallback — build locally:**
```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64   # once, enables cross-build
docker buildx build --platform linux/arm64 \
  -f deploy/docker/serving.Dockerfile \
  -t ghcr.io/<owner>/mlops-drift:latest --push .
```

## 3. Set the Grafana password
- `deploy/helm/kps-values-oke.yaml` → set `grafana.adminPassword` to a real strong password.
- If you use a different registry/owner, update `images[0]` in `deploy/k8s/oke/kustomization.yaml`.

## 4. Install monitoring (Grafana gets the public LB)
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kps prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f deploy/helm/kps-values-oke.yaml
```

## 5. Deploy the app + monitors + dashboards
```bash
# app workloads (image-swapped, PVC sized for OCI):
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/k8s/oke | kubectl apply -f -
# scrape configs + dashboards (need monitoring ns + ServiceMonitor CRD from step 4):
kubectl apply -f deploy/k8s/servicemonitors.yaml \
               -f deploy/k8s/grafana-dashboard-serving.yaml \
               -f deploy/k8s/grafana-dashboard-drift.yaml

kubectl -n mlops wait --for=condition=complete job/seed-train --timeout=600s
kubectl -n mlops get pods            # serving + controller 1/1
```

## 6. Get the live link + drive the demo
```bash
kubectl -n monitoring get svc kps-grafana -w     # wait for EXTERNAL-IP (OCI LB, ~1-2 min)
```
`http://<EXTERNAL-IP>` → log in `admin` / your password → **Drift & Retrain** dashboard.

Trigger the recovery so the dashboard has a story:
```bash
kubectl -n mlops exec deploy/controller -- make replay
```
`realized_f1` dips ~0.28 → recovers ~0.83, `monitor_model_version` ticks 1 → 2. That panel is
the link to share.

## 7. Secure before sharing
- Grafana admin password is set (step 3) and anon is off — keep it that way.
- Optional domain + TLS: point a DNS A record at the LB IP and front Grafana with cert-manager +
  an Ingress, or terminate TLS on the OCI LB.
- Quick-create security lists already allow the LB → node ports. With a custom VCN, open the
  node-pool NSG to the LB health-check + nodePort range.

## 8. Teardown (stop the meter)
Always-Free shouldn't bill, but to fully clean up:
```bash
kubectl delete -f deploy/k8s/servicemonitors.yaml --ignore-not-found
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/k8s/oke | kubectl delete -f -
helm uninstall kps -n monitoring        # also releases the LB + its block volumes
# then delete the node pool + cluster in the Console
```
The Grafana LoadBalancer and any PVC block volumes are released by deleting their Services/PVCs —
check the Console (Networking → Load Balancers, Storage → Block Volumes) so none linger.
