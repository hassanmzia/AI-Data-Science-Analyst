# Build stage
FROM node:20-alpine as build

WORKDIR /app

ARG VITE_API_URL=http://172.168.1.95:8050
ARG VITE_MCP_URL=http://172.168.1.95:4050
ARG VITE_WS_URL=ws://172.168.1.95:8050/ws

ENV VITE_API_URL=$VITE_API_URL
ENV VITE_MCP_URL=$VITE_MCP_URL
ENV VITE_WS_URL=$VITE_WS_URL

COPY package*.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3050

CMD ["nginx", "-g", "daemon off;"]
