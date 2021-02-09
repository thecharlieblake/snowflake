FROM tensorflow/tensorflow

# Command line apps required
RUN apt update && apt install -y wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev graphviz graphviz-dev patchelf

# Download MuJoCo
RUN wget https://www.roboti.us/download/mjpro150_linux.zip \
    && mkdir -p /root/.mujoco/ \
    && unzip mjpro150_linux.zip -d /root/.mujoco/ \
    && rm mjpro150_linux.zip
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
# Copy mujoco key (required in base dir of repo) to required location
COPY ./mjkey.txt /root/.mujoco/mjkey.txt

# Install rest of dependencies
RUN pip install pip --upgrade
RUN pip install num2words
RUN pip install numpy
RUN pip install sacred
RUN pip install pandas
RUN pip install beautifulsoup4
RUN pip install termcolor
RUN pip install sklearn
RUN pip install cffi
RUN pip install Cython
RUN pip install lockfile
RUN pip install glfw
RUN pip install imageio
RUN pip install gym
RUN pip install mujoco-py==1.50.1.68
RUN pip install lxml

WORKDIR /home/snowflake