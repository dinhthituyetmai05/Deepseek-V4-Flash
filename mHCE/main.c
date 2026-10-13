#include <stdio.h>
#include "include/datapath_mHCE.h"

void print_size(const char *name, size_t bytes)
{
    printf("\t %s: ", name);

    if (bytes >= 1024ULL * 1024 * 1024) {
        printf("%.2f GiB\n",
               (double)bytes / (1024.0 * 1024 * 1024));
    }
    else if (bytes >= 1024ULL * 1024) {
        printf("%.2f MiB\n",
               (double)bytes / (1024.0 * 1024));
    }
    else if (bytes >= 1024) {
        printf("%.2f KiB\n",
               (double)bytes / 1024.0);
    }
    else {
        printf("%zu bytes\n", bytes);
    }
}

int main()
{
    printf("1. Xác định kích thước các buffer \n");
    print_size("DDR4", sizeof(DDR4));
    print_size("WTB", sizeof(WTB));
    print_size("RAB", sizeof(RAB));

    print_size("X_l", sizeof(RAB.i_x));

    return 0;
}