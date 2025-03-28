FROM public.ecr.aws/lambda/python:3.12
RUN microdnf -y install git binutils gcc && microdnf -y clean all
