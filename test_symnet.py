from run_symnet import *


def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    parser = make_parser()
    args = parser.parse_args()
    utils.display_args(args, logger)


    # logging and pretrained weight dirs
    log_dir = osp.join(cfg.LOG_ROOT_DIR, args.name)


    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, 'test', 
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    

    logger.info("Loading network")
    network_module = importlib.import_module('models.'+args.network)
    model = network_module.Model(train_dataloader.dataset, args).cuda()
    print(model)

    


    # initialization (model weight, optimizer, lr_scheduler, clear logs)

    if args.trained_weight is None:
        raise ValueError("Need to specify checkpoint path")
    else:
        # load weight
        checkpoint = torch.load(args.trained_weight)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Checkpoint <= "+args.trained_weight)


    # evaluator
    evaluator = CZSL_Evaluator(test_dataloader.dataset, model)


    # trainval
    logger.info('Start evaluation')
    with torch.no_grad():
        current_report = test_epoch(model, evaluator, test_dataloader, writer, epoch)

    # print test results
    print("Current: " + utils.formated_czsl_result(current_report))
    logger.info('Finished.')



if __name__=="__main__":
    main()
