import logging.config
from spl import ProjectLoader, TopicModeling

log = None


def main():
    log.info("main")

    pl = ProjectLoader()
    # pl.load(r'D:\Projects\Java\javaparser-no-test')
    pl.load(r'D:\SPLGroup\spl-workspaces\java\cocome-maven-project')

    tp = TopicModeling()
    tp.analyze(pl.home, pl.files)


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
