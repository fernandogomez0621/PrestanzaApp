# --- Etapa 1: build de la app React ---
FROM node:20-alpine AS build
# MODO=indices compila la versión que solo tiene los modelos por índices (puerto 8502)
ARG MODO=completo
ENV VITE_MODO=$MODO
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# --- Etapa 2: servir con Nginx ---
FROM nginx:alpine
ARG MODO=completo
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf nginx.indices.conf /tmp/
RUN if [ "$MODO" = "indices" ]; then cp /tmp/nginx.indices.conf /etc/nginx/conf.d/default.conf; \
    else cp /tmp/nginx.conf /etc/nginx/conf.d/default.conf; fi
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
