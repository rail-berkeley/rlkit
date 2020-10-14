FROM mdalal/carbongym_base

COPY nvidia_icd.json /usr/share/vulkan/icd.d

COPY nvidia_driver_setup_450_80_02.sh /nvidia_driver_setup_450_80_02.sh
RUN  /nvidia_driver_setup_450_80_02.sh
