import logging
import stdlib.jsonx as json
from iplan import IPlanObjectModel


def training(request):
    LOG = logging.getLogger('training')
    LOG.info("Start training ...")

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    # ip:port/db
    # user/password

    # areaFeature  -> area_id
    # skillFeature -> skill_id
    # id           -> time_series_id   (tb_ipr_conf_master)
    # iDataId      -> data_master_id

    #
    # Convert the "strange" parameter names passed in the call info
    # more 'intelligent' names

    ds_info = {
        "drivername": "postgresql",
        "username": body['user'],
        "password": body['password'],
        "host": body['ip'],
        "port": body['port'],
        "database": body['db'],
    }

    area_id  = body['areaFeature']
    skill_id = body['skillFeature']
    time_series_id = body['id']
    data_master_id = body['iDataId']
    plan_id = body['planId']

    ipom = IPlanObjectModel(ds_info)
    with ipom.connect():
        ts = ipom.time_series().focussed(time_series_id).using_data_master(data_master_id)

        df = ts.train().select(area=area_id, skill=skill_id, new_format=False)


    pass