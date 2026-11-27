# Solver provenance

The solver under this directory comes from Jim's supplied twstab 2.18.0 source archive. The archive SHA-256 is `fe1520170f6a5ad8b8904441772c8045aa5935cfa5e8a6eef78b4f01c0d7ce66`. Jim approved public inclusion in the generator repository.

The integrated version adds the exact conservative profile entry, the continued A5 origin contour, and the fixed-speed unfolding interface. The original solver README and design document describe its broader standalone capabilities. [Integration interface and validation](../docs/jim-a5-unfolding.md)

The generator repository also adds a temporal Floquet action command for F2. This local extension uses the bundled version number for the original solver and versions its orbit, request and response formats separately. [Temporal Floquet integration](../docs/twstab-floquet.md)
