# 使用官方 Node.js 18 LTS 镜像作为基础
FROM node:18-alpine

# 设置工作目录
WORKDIR /usr/src/app

# 复制 package.json 和 package-lock.json (如果存在)
COPY package*.json ./

# 安装项目依赖
RUN npm install

# 复制应用源代码
COPY . .

# 暴露应用程序使用的端口
EXPOSE 3000

# 定义环境变量 (可以在 docker-compose 中覆盖)
ENV PORT=3000
# FAL_KEY 应该在运行时通过 docker-compose 传入，而不是硬编码在这里

# 运行应用程序的命令
CMD [ "npm", "start" ]
