FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf update 
RUN dnf install vim bash tcsh -y 

ENV MANPATH="/usr/local/share/man:/usr/share/man"
ENV PATH="/opt/local/bin:$PATH"

WORKDIR /tmp


#  Install Miniconda. 

COPY miniconda3.sh ./
RUN bash miniconda3.sh -b -p /opt/miniconda3.9
RUN rm miniconda3.sh
ENV PATH="/opt/miniconda3.9/bin:$PATH"

#  Install tropicalwidth package. 

ADD tropicalwidth-1.0.0.tar .
RUN pip install ./tropicalwidth 
RUN rm -fr tropicalwidth

ENV OCEANMASK="/opt/share/oceanmask.nc"
RUN mkdir -p /opt/share
COPY oceanmask.nc ${OCEANMASK}

#  List of OSCAR files. 

COPY oscar_download.txt /opt/share/
COPY oscar_download /opt/local/bin/

# USER user
# WORKDIR /home/user

CMD ["/bin/bash"]
