param([string]$Release = "anomaly")
helm uninstall $Release
minikube stop
