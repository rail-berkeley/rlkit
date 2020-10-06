FROM mdalal/carbongym_base

COPY nvidia_icd.json /usr/share/vulkan/icd.d

COPY nvidia_driver_setup.sh /nvidia_driver_setup_418_43.sh
RUN  /nvidia_driver_setup.sh
