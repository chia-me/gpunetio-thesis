# Dipendenze di sistema

Queste dipendenze vanno installate sul server prima di compilare.
Non sono nel repository perché sono librerie di sistema, non codice del progetto.

## DOCA SDK (librerie, non header)
Gli header DOCA sono già nel repo sotto `vendor/doca/include/`.
Le librerie `.so` e `.a` devono essere installate:
- Pacchetti: `doca-sdk`, `doca-runtime` da NVIDIA DOCA repository
- Path atteso: `/opt/mellanox/doca/lib/x86_64-linux-gnu/`

## CUDA Toolkit
- Versione usata: 13.2
- Path atteso: `/usr/local/cuda-13.2/`
- Download: https://developer.nvidia.com/cuda-downloads

## GDRCopy
Necessario per il kernel driver `gdrdrv` che permette alla NIC di fare DMA
direttamente nella memoria GPU (usato internamente da DOCA GPUNetIO).
Non va chiamato direttamente dal codice applicativo.

- Repository: https://github.com/NVIDIA/gdrcopy
- Versione usata: 2.5
- Installazione:
    git clone https://github.com/NVIDIA/gdrcopy.git ~/gdrcopy
    cd ~/gdrcopy
    make
    sudo ./insmod.sh

## Mellanox OFED / MLNX_OFED
Driver per la NIC BlueField-2 (ConnectX-6 Dx).
- Download: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
