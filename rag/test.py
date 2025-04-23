#!/usr/bin/env python3
import time
import subprocess
def foo():
	# Instructions to be evaluated.
    ...
foo()

def get_cpu_power():
    """Récupère la puissance consommée par le CPU en watts"""
    result = subprocess.run(
        ["cat", "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"],
        stdout=subprocess.PIPE, text=True
    )
    print(result.stdout)
    return int(result.stdout.strip())

def monitor_cpu_power(energy_log):
    """Surveille la consommation énergétique du CPU en arrière-plan"""
    while energy_log["running"]:
        power = get_cpu_power()  # Récupère la consommation actuelle (W)
        timestamp = time.time()  # Enregistre le temps actuel
        energy_log["Time"].append(timestamp)
        energy_log["Power"].append(power)
        time.sleep(0.1)  # Mesure toutes les 100 ms
