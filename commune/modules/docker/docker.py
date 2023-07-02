import commune as c
import os
import pandas as pd
import json

class Docker(c.Module): 

    def ps(sudo=False):
        return c.cmd('docker ps -a', sudo=sudo)

    def ps():
        return c.cmd('sudo docker ps -a')
    
    @classmethod
    def dockerfile(cls, path = c.repo_path): 
        path =  [f for f in c.ls(path) if f.endswith('Dockerfile')][0]
        return c.get_text(path)
    
    @classmethod
    def resolve_repo_path(cls, path):
        if path is None:
            path = c.repo_path
        else:
            path = c.repo_path + '/' + path
        return path

    @classmethod
    def resolve_docker_compose_path(cls,path = None):
        path = cls.resolve_repo_path(path)
        return [f for f in c.ls(path) if 'docker-compose' in os.path.basename(f)][0]
        return path

    @classmethod
    def docker_compose(cls, path = c.repo_path): 
        docker_compose_path = cls.resolve_docker_compose_path(path)
        return c.load_yanl(docker_compose_path)
    
    @classmethod
    def build(cls, path = None, tag = None, sudo=True):
        path = cls.resolve_repo_path(path)
        return c.cmd(f'docker-compose build', sudo=sudo, cwd=path)
    
    
    @classmethod
    def containers(cls,  sudo:bool = True):
        data = [f for f in c.cmd('docker ps -a', sudo=sudo).split('\n')[1:]]
        def parse_container_info(container_str):
            container_info = {}
            fields = container_str.split()

            c.print(fields)
            container_info['container_id'] = fields[0]
            container_info['image'] = fields[1]
            container_info['command'] = fields[2]
            container_info['created'] = fields[3] + ' ' + fields[4]
            container_info['status'] = ' '.join(fields[5:fields.index('ago') + 1])
            container_info['ports'] = ' '.join(fields[fields.index('ago') + 2:-1])
            container_info['name'] = fields[-1]

            return container_info

        
        return [parse_container_info(container_str) for container_str in data if container_str]
