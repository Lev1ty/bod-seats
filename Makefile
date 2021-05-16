SHELL=/bin/bash
include .env
export
gpustat:
	gpustat -cupF -i 1
jupyter:
	jupyter nbextension enable --py widgetsnbextension
	jupyter notebook --ip $$PUBLIC_IP
