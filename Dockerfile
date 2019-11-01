FROM tiangolo/python-machine-learning:python3.6

LABEL version="1.0"
LABEL description="Dev Environment for McMaster University, SFWRTECH 4NN3 Course"

# Update Conda
RUN conda update -n base -c defaults conda

# Install ML Libraries  
RUN conda install -y tensorflow
RUN conda install -y keras
RUN conda install -y numpy
RUN conda install -y scipy
RUN conda install -y scikit-learn
RUN conda install -y pandas
RUN conda install -y matplotlib
RUN conda install -y pylint

# Creating New User and Set Password
RUN useradd --create-home --shell /bin/bash yury.stanev
RUN echo 'yury.stanev:passw0rd' | chpasswd

# Set User & WORKDIR
USER yury.stanev
ENV HOME /home/yury.stanev
WORKDIR /home/yury.stanev