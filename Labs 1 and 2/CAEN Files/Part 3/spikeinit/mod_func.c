#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _kd3h5_reg();
extern void _na3h5_reg();

modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," kd3h5.mod");
fprintf(stderr," na3h5.mod");
fprintf(stderr, "\n");
    }
_kd3h5_reg();
_na3h5_reg();
}
