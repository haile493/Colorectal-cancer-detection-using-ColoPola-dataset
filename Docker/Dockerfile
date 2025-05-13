##### Dockerfile #####
# Use python 3.9
FROM python:3.9

# Thiết lập working directory
WORKDIR /colocancer

# Copy file requirements.txt vào trong image
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install -r requirements.txt

# Copy source code vào trong image
COPY . .
