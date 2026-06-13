# ==========================================
# Cardio AI Assistant - Optimized Frontend Dockerfile
# ==========================================

# Stage 1: Dependency Installation & Build
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline --no-audit

# Copy application code
COPY . .

# Build the Vite application
RUN npm run build

# Stage 2: Production Server
FROM nginx:alpine-slim

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Optionally, add custom nginx configuration
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]