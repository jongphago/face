import torch


def train(
    dataloader,
    loss,
    model,
    optimizer,
    lr_scheduler,
    config,
):
    model.to_train()

    for index, data in enumerate(dataloader):
        embeddings = model.embedding(data.image.cuda())
        total_loss = 0
        if config.is_fr:
            # face_pred = model.face(embeddings)
            # fr_loss: torch.Tensor = loss.face(face_pred, data)
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

        if index % 10 == 0:
            print(f"{index:4d},", end=" ")
            if config.is_fr:
                print(f"fr: {fr_loss.item():4.2f}", end=" ")
            if config.is_ae:
                print(
                    f"age: {age_loss.item():4.2f}, age_group: {age_group_loss.item():4.2f}, age_mean_var: {weighted_mean_variance_loss.item():4.2f}",
                    end=" ",
                )
            if config.is_kr:
                print(f"kinship: {kinship_loss.item():4.2f}", end=" ")
            print(f"TOTAL_LOSS: {total_loss}")

        torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), 5)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # break
