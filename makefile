build:
	docker build -t memory .

start:
	docker run -d --name memory --ipc=host --gpus all -p 5000:5000 -p 8888:8888 --rm -v "/home/gianluca/git/memory:/workspace" -v "/data:/data" -t memory

attach:
	docker exec -it memory /bin/zsh

stop:
	docker kill memory

clean:
	docker system prune -a && \
	docker image prune && \
	docker volume prune
