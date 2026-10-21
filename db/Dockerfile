FROM alpine:3.20
RUN apk add --no-cache sqlite
COPY init.sh /usr/local/bin/oracle-db-init
ENTRYPOINT ["sh", "/usr/local/bin/oracle-db-init"]
