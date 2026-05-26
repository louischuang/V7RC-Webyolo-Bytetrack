FROM node:24-alpine

WORKDIR /app

RUN apk add --no-cache ffmpeg python3 py3-pip \
  && pip3 install --break-system-packages --no-cache-dir yt-dlp

COPY package.json ./
COPY server.mjs ./

ENV STREAM_GATEWAY_PORT=3010
ENV STREAM_OUTPUT_DIR=/var/lib/v7rc-streams
ENV STREAM_DEFAULT_OUTPUT=mjpg

RUN mkdir -p /var/lib/v7rc-streams

EXPOSE 3010

CMD ["node", "server.mjs"]
