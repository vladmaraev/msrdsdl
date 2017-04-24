FROM nvidia/cuda:8.0-cudnn5-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz emacs-nox && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh && \
    #    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-4.3.11-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.3.11-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.3.11-Linux-x86_64.sh

RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

COPY . /src
WORKDIR /src

# Python
ARG python_version=3.6.1
RUN conda install -y python=${python_version} && \
    pip install tensorflow && \
    pip install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    conda install Pillow scikit-learn notebook pandas matplotlib nose pyyaml six h5py && \
    pip install git+git://github.com/fchollet/keras.git && \
    conda clean -yt

ENV PYTHONPATH='/src/:$PYTHONPATH'


RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt
