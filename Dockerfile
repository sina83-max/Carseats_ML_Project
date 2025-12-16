FROM ubuntu:latest
LABEL authors="dev-nazari"

ENTRYPOINT ["top", "-b"]