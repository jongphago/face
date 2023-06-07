import torch
from datetime import datetime


def train(
    dataloader,
    loss,
    model,
    optimizer,
    lr_scheduler,
    config,
    logger,
    epoch,
    global_step,
):
    model.to_train()
    for data in dataloader:
        global_step.increment()
        index = global_step.get()
        embeddings = model.embedding(data.image.cuda())
        loss_list = []
        loss_name = []
        if config.is_fr:
            fr_loss = loss.module_partial_fc(embeddings, data.face_label.cuda())
            loss_list.append(fr_loss)
            loss_name.append("fr_loss")
        if config.is_ae:
            age_pred, age_group_pred = model.age(embeddings)
            total_age_loss = loss.age((age_pred, age_group_pred), data)
            age_loss, age_group_loss, weighted_mean_variance_loss = total_age_loss
            loss_list += [age_loss, age_group_loss, weighted_mean_variance_loss]
            loss_name += ["age_loss", "age_group_loss", "weighted_mean_variance_loss"]
        if config.is_kr:
            kinship_pred = model.kinship(embeddings)
            kinship_loss = loss.kinship(kinship_pred, data)
            loss_list.append(kinship_loss)
            loss_name.append("kinship_loss")

        total_loss = loss.multi_loss_layer(loss_list)

        if index % config.log_interval == 0:
            now = datetime.now().strftime("%D %T")
            num_data = len(dataloader)
            print(
                f"{now} Epoch:{epoch}/{config.num_epoch} {index+num_data*(epoch-1):4d}/{num_data*config.num_epoch}({(index+num_data*(epoch-1))/(num_data*config.num_epoch):2.0%}),",
                end=" ",
            )
            log_dict = {
                "step": index,
                "Train/epoch": epoch,
                "Train/learning rate": lr_scheduler.get_last_lr()[0],
            }

            for value, name in zip(loss_list, loss_name):
                log_dict[f"Train/{name}"] = value
                print(f"{name}: {value:4.2f}", end=" ")
            print(f"w.total_loss:{total_loss:4.2f}")
            log_dict["Train/total_loss"] = total_loss.item()
            if logger:
                logger.log(log_dict)
        torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), 5)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # if index % 20 == 0:
        # break
