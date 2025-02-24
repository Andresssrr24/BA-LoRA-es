import torch
import torch.nn.functional as F

def consistency_regularization(pretrained_logits, fine_tuned_logits):
    # Consistency Regularization (KL Divergence)
    fine_tuned_log_probs = F.log_softmax(fine_tuned_logits, dim=-1)
    pretrained_probs = F.softmax(pretrained_logits, dim=-1)
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    return loss_fn(fine_tuned_log_probs, pretrained_probs)


def diversity_regularization(logits):
    # Diversity Regularization (Entropy)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * log_probs, dim=-1))


def svd_regularization(logits, k=10, n_iter=2):
    # SVD Regularization (Randomized SVD)
    if logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])  # Flatten to [B*L, N]
    logits = (logits - torch.mean(logits, dim=-1, keepdim=True)) / torch.std(logits, dim=-1, keepdim=True)
    _, num_features = logits.shape
    random_matrix = torch.randn(num_features, k, device=logits.device)
    projected_matrix = logits @ random_matrix
    for _ in range(n_iter):
        projected_matrix = logits @ (logits.T @ projected_matrix)
    q, _ = torch.linalg.qr(projected_matrix)
    b = q.T @ logits
    _, s, _ = torch.linalg.svd(b, full_matrices=False)
    s = s / torch.sum(s)  # Normalize singular values
    top_k_singular_values = s[:k]
    total_singular_values = torch.sum(s)
    loss = -torch.sum(top_k_singular_values**2) / torch.sum(total_singular_values**2) if total_singular_values > 0 else torch.tensor(0.0, device=logits.device)
    return loss


def compute_regularization_losses(pretrained_logits, fine_tuned_logits, logits, lambda_cr, lambda_dr, lambda_svdr, svd_k=5):
    # Combine all regularization losses
    cr_loss = consistency_regularization(pretrained_logits, fine_tuned_logits)
    dr_loss = diversity_regularization(logits)
    svdr_loss = svd_regularization(logits, svd_k)
    total_regularization_loss = lambda_cr * cr_loss + lambda_dr * dr_loss + lambda_svdr * svdr_loss
    return {
        "total_regularization_loss": total_regularization_loss,
        "cr_loss": cr_loss,
        "dr_loss": dr_loss,
        "svdr_loss": svdr_loss,
    }
