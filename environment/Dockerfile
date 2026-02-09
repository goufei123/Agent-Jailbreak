FROM continuumio/miniconda3:latest

# 让后续 RUN 用 bash，便于 conda activate
SHELL ["/bin/bash", "-lc"]

WORKDIR /app

# 尽量先装系统依赖再拷贝大项目，可以更好利用缓存
RUN apt-get update -y && apt-get install -y xxd wget curl \
    && rm -rf /var/lib/apt/lists/*

# 创建固定 Python 版本的环境（3.11 也可以改成 3.10）
ARG PYVER=3.10
RUN conda create -y -n app python=${PYVER} && conda clean -afy
RUN echo "conda activate app" >> ~/.bashrc

COPY . /app
COPY usrfile.txt /usr
COPY sensitive_info_root.txt /root

RUN chmod +x init.sh start_processes.sh

RUN conda activate app \
 && pip install --upgrade pip setuptools wheel \
 && pip install  \
      "flask>=3.0,<4" \
      "jinja2>=3.1" \
      "werkzeug>=3.0" \
      "click>=8.1" \
      "itsdangerous>=2.2" \
      requests \
      pyyaml \
      jsonpickle \
      psutil \
      pyperclip \
      pillow \
      watchdog

EXPOSE 5758
EXPOSE 5389

ENTRYPOINT ["/app/init.sh"]
