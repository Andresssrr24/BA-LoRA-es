import torch
import torch.nn.functional as F

# Consistency Regularization
def consistency_regularization(pretrained_logits, fine_tuned_logits):
    """
    Consistency regularization based on KL divergence.
    """
    fine_tuned_log_probs = F.log_softmax(fine_tuned_logits, dim=-1)
    pretrained_probs = F.softmax(pretrained_logits, dim=-1)
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    return loss_fn(fine_tuned_log_probs, pretrained_probs)

# Diversity Regularization
def diversity_regularization(logits):
    """
    Diversity regularization based on entropy.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * log_probs, dim=-1))

# SVD Regularization
def svd_regularization(logits, k=5, n_iter=2):
    """
    Regularization using randomized SVD.
    """
    if logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])

    _, num_features = logits.shape
    random_matrix = torch.randn(num_features, k, device=logits.device)
    projected_matrix = logits @ random_matrix

    for _ in range(n_iter):
        projected_matrix = logits @ (logits.T @ projected_matrix)

    q, _ = torch.linalg.qr(projected_matrix)
    b = q.T @ logits
    _, s, _ = torch.linalg.svd(b, full_matrices=False)

    top_k_singular_values = s[:k]
    total_singular_values = torch.sum(s)
    return -torch.sum(top_k_singular_values) / total_singular_values

# Compute total regularization loss
def compute_regularization_losses(
    pretrained_logits, fine_tuned_logits, logits, lambda_cr, lambda_dr, lambda_svdr, svd_k=5
):
    """
    Compute and return the total regularization loss.
    """
    cr_loss = consistency_regularization(pretrained_logits, fine_tuned_logits)
    dr_loss = diversity_regularization(logits)
    svdr_loss = svd_regularization(logits, svd_k)

    total_regularization_loss = lambda_cr * cr_loss + lambda_dr * dr_loss + lambda_svdr * svdr_loss

    return total_regularization_loss

