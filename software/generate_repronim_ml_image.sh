docker run kaczmarj/neurodocker:0.6.0 generate docker --base=ubuntu:18.04 \
--pkg-manager=apt \
--install git nano unzip \
--miniconda \
    version=latest \
    create_env='repronim_ml' \
    activate=true \
    conda_install="python=3.10 numpy=1.22 pandas=1.4 scikit-learn=1.0.2 seaborn=0.11" \
    pip_install="tensorflow==2.8 datalad[full]==0.15.6" \
--add-to-entrypoint "source activate repronim_ml" \
--entrypoint "/neurodocker/startup.sh  python" > Dockerfile