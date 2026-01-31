# Build stage
FROM node:20-alpine as build

WORKDIR /app

ARG REACT_APP_API_URL=http://172.168.1.95:8050
ARG REACT_APP_MCP_URL=http://172.168.1.95:4050
ARG REACT_APP_WS_URL=ws://172.168.1.95:8050/ws

ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_MCP_URL=$REACT_APP_MCP_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 3050

CMD ["nginx", "-g", "daemon off;"]
