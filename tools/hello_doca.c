#include <stdio.h>
#include <string.h>

#include <doca_dev.h>
#include <doca_error.h>

int main()
{
    struct doca_devinfo **list;
    uint32_t nb_devs;
    doca_error_t res;
    uint32_t i;
    char pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE];
    char iface_name[DOCA_DEVINFO_IFACE_NAME_SIZE];
    char ibdev_name[DOCA_DEVINFO_IBDEV_NAME_SIZE];

    res = doca_devinfo_create_list(&list, &nb_devs);

    if (res != DOCA_SUCCESS) {
        printf("Errore: %s\n",
               doca_error_get_descr(res));
        return -1;
    }

    printf("Device trovati: %u\n\n", nb_devs);

    /* Itera su ogni dispositivo e stampa le sue informazioni */
    for (i = 0; i < nb_devs; i++) {
        printf("=== Device %u ===\n", i);

        /* PCI Address */
        res = doca_devinfo_get_pci_addr_str(list[i], pci_addr);
        if (res == DOCA_SUCCESS) {
            printf("  PCI Address: %s\n", pci_addr);
        }

        /* Interface Name */
        res = doca_devinfo_get_iface_name(list[i], iface_name, sizeof(iface_name));
        if (res == DOCA_SUCCESS) {
            printf("  Interface Name: %s\n", iface_name);
        }

        /* InfiniBand Device Name */
        res = doca_devinfo_get_ibdev_name(list[i], ibdev_name, sizeof(ibdev_name));
        if (res == DOCA_SUCCESS) {
            printf("  IB Device Name: %s\n", ibdev_name);
        }

        printf("\n");
    }

    doca_devinfo_destroy_list(list);

    return 0;
}
