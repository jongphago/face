from math import floor
import numpy as np


def get_folder_name(family_id: str) -> str:
    """_summary_

    Args:
        family_id (str): 접두사 F를 포함하는 네자리 숫자 문자열, 'F0###' 형식

    Returns:
        str: prefix를 제거한 네자리 숫자 문자열. 동일한 백의 자리를 같는 수 중
        가장 작은 자연수. '0#00' 형식
    """
    return f"{floor(int(family_id[-4:])/100)*100:04d}"


def tensor_to_int(x):
    return x.cpu().data.numpy().item()


def log_verification_output(validate_output, wandb_logger, prefix, step):
    best_distances, (accuracy, precision, recall, roc_auc, tar, far) = validate_output
    if wandb_logger:
        wandb_logger.log(
            {
                "step": step,
                f"{prefix}/accuracy": np.mean(accuracy),
                f"{prefix}/precision": np.mean(precision),
                f"{prefix}/recall": np.mean(recall),
                f"{prefix}/best_distances": np.mean(best_distances),
                f"{prefix}/tar": np.mean(tar),
                f"{prefix}/far": np.mean(far),
                f"{prefix}/roc_auc": roc_auc,
            }
        )


def add_parameters(config, model, loss):
    params = [{"params": model.embedding.parameters()}]
    if config.is_fr:
        params.append({"params": model.face.parameters()})
        params.append({"params": loss.module_partial_fc.parameters()})
    if config.is_ae:
        params.append({"params": model.age.parameters()})
    if config.is_kr:
        params.append({"params": model.kinship.parameters()})
    params.append({"params": loss.multi_loss_layer.parameters()})
    return params
