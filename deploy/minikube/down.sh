#!/usr/bin/env bash
helm uninstall "${1:-anomaly}"
minikube stop
