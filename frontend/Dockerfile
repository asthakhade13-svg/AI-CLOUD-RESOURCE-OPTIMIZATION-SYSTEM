FROM nginx:alpine

# Remove default configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy reverse proxy configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/

# Copy dashboard static file
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
