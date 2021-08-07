def run_train(net, args, trainloader, device):
    
    criterion_edge = nn.CrossEntropyLoss()
    optimizer_edge = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        trainloss = train(epoch, net, criterion_edge, optimizer_edge, trainloader, device)
        acc, best_acc = test(epoch, net, criterion_edge, testloader, device)
        logger.debug(f"The result is: {acc}")
        write_csv('acc_' + args.workspace + '_' + strtime + '.csv', str(acc))
        list_loss.append(trainloss)
    logger.debug("===> BEST ACC. PERFORMANCE: %.2f%%" % (best_acc))
