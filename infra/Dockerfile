FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


#set working directory
WORKDIR /app

#copy the requirements file into container
COPY requirements.txt /app/


COPY kosmos.py /app/kosmos.py




#install the basic requirements
RUN pip install --no-cache-dir -r requirements.txt


#install git
RUN apt-get update && apt-get install -y git


#install Nividia apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


#install torchsacle
RUN git clone https://github.com/microsoft/torchscale.git && \
    cd torchscale && \
    pip install -e .

#set the environment variables
# Set environment variables for AWS credentials


#copy the train_kosmos.py /app/
COPY train_kosmos.py /app/

#run the training script
CMD ["python3", "train_kosmos.py"]









# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


# #set working directory
# WORKDIR /app

# #copy the requirements file into container
# COPY requirements.txt /app/

# COPY kosmos.py /app/kosmos.py


# # COPY kosmos.py /app/kosmos.py



# #install the basic requirements
# RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install boto3

# RUN apt-get update && apt-get install -y git

# #install Nividia apex
# RUN git clone https://github.com/NVIDIA/apex && \
#     cd apex && \
#     pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# #install torchsacle
# RUN git clone https://github.com/microsoft/torchscale.git && \
#     cd torchscale && \
#     pip install -e .

# #set the environment variables
# # Set environment variables for AWS credentials

# #copy the train_kosmos.py /app/
# COPY train_kosmos.py /app/

# #run the training script
# CMD ["python", "train_kosmos.py"]
