import torch


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
        total_loss = 0
        if config.is_fr:
            fr_loss = loss.module_partial_fc(embeddings, data.face_label.cuda())
            total_loss += fr_loss * config.weight.face
        if config.is_ae:
            age_pred, age_group_pred = model.age(embeddings)
            total_age_loss = loss.age((age_pred, age_group_pred), data)
            age_loss, age_group_loss, weighted_mean_variance_loss = total_age_loss
            total_loss += sum(
                [
                    age_loss * config.weight.age,
                    age_group_loss * config.weight.age_group,
                    weighted_mean_variance_loss * config.weight.age_mean_var,
                ]
            )
        if config.is_kr:
            kinship_pred = model.kinship(embeddings)
            kinship_loss = loss.kinship(kinship_pred, data)
            total_loss += kinship_loss * config.weight.kinship

        if index % config.log_interval == 0:
            print(f"{index:4d},", end=" ")
            log_dict = {
                "step": index,
                "Train/epoch": epoch,
                "Train/learning rate": lr_scheduler.get_last_lr()[0],
            }
            if config.is_fr:
                print(f"fr: {fr_loss.item():4.2f}", end=" ")
                log_dict["Train/fr_loss"] = fr_loss.item()
            if config.is_ae:
                print(
                    f"age: {age_loss.item():4.2f}, age_group: {age_group_loss.item():4.2f}, age_mean_var: {weighted_mean_variance_loss.item():4.2f}",
                    end=" ",
                )
                log_dict["Train/age_loss"] = age_loss.item()
                log_dict["Train/age_group_loss"] = age_group_loss.item()
                log_dict[
                    "Train/weighted_mean_variance_loss"
                ] = weighted_mean_variance_loss.item()
            if config.is_kr:
                print(f"kinship: {kinship_loss.item():4.2f}", end=" ")
                log_dict["Train/kinship_loss"] = kinship_loss.item()
            print(f"WEIGHTED_TOTAL_LOSS: {total_loss}")
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
