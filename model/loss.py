import torch
import torch.nn as nn

class ComplexLoss(nn.Module):
    def __init__(self):
        super(ComplexLoss, self).__init__()

    def forward(self, r1, theta1, r2, theta2):
        loss = r1**2 + r2**2 - 2*r1*r2*torch.cos(theta2 - theta1)
        return loss.mean()


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        #self.complex_loss = ComplexLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.epolen_bins = nn.Parameter(
                torch.linspace(0.0024999999999995026, 0.02400000000000002, 257 - 1),
                requires_grad=False,
            )

            # log_d_predictions,
            # mel_prediction,
            # phase_prediction,
            # epochlen_prediction,
            # d_rounded,
            # text_masks,
            # acoustic_masks,
            # text_lens,
            # acoustic_lens,

    def forward(self, inputs, predictions):
        (
            _,
            _, 
            _,
            _,
            text_lens,
            max_text_len,
            mel_targets,
            phase_targets,
            acoustic_lens,
            max_acoustic_len,
            epochdur_targets,
            epochlen_targets,
        ) = inputs
        
        (
            log_epochdur_predictions,
            mel_predictions,
            phase_predictions, 
            epochlen_predictions,
            _,
            text_masks,
            acoustic_masks,
            _,
            _,
        ) = predictions
        
        epochlen_predictions = epochlen_predictions.squeeze(-1)
        
        mel_targets = mel_targets.transpose(1, 2)
        phase_targets = phase_targets.transpose(1, 2)
        
        text_masks = ~text_masks
        acoustic_masks = ~acoustic_masks
        
        log_epochdur_targets = torch.log(epochdur_targets.float())
        
        mel_targets = mel_targets[:, : acoustic_masks.shape[1], :]
        acoustic_masks = acoustic_masks[:, :acoustic_masks.shape[1]]

        log_epochdur_targets.requires_grad = False
        mel_targets.requires_grad = False
        phase_targets.requires_grad = False
        epochlen_targets.requires_grad = False

        log_epochdur_targets = log_epochdur_targets.masked_select(text_masks)
        log_epochdur_predictions = log_epochdur_predictions.masked_select(text_masks)

        mel_predictions = mel_predictions.masked_select(acoustic_masks.unsqueeze(-1))
        phase_predictions = phase_predictions.masked_select(acoustic_masks.unsqueeze(-1))
        
        mel_targets = mel_targets.masked_select(acoustic_masks.unsqueeze(-1))
        phase_targets = phase_targets.masked_select(acoustic_masks.unsqueeze(-1))
        
        epochlen_predictions = epochlen_predictions.masked_select(acoustic_masks.unsqueeze(-1)).reshape(-1, 256)
        epochlen_targets = epochlen_targets.masked_select(acoustic_masks)
        epochlen_targets_bucket = torch.bucketize(epochlen_targets, self.epolen_bins)

        mel_loss_l1 = self.mae_loss(mel_predictions, mel_targets)
        mel_loss_l2 = self.mse_loss(mel_predictions, mel_targets)
        
        phase_loss_l1 = self.mae_loss(phase_predictions, phase_targets) / 50.0
        phase_loss_l2 = self.mse_loss(phase_predictions, phase_targets) / 50.0
        
        # com_loss = self.complex_loss(mel_targets, phase_targets, mel_predictions, phase_predictions)

        duration_loss_l1 = self.mae_loss(log_epochdur_predictions, log_epochdur_targets)
        duration_loss_l2 = self.mse_loss(log_epochdur_predictions, log_epochdur_targets)
        
        # length_loss_l1 = self.mae_loss(epochlen_predictions, epochlen_targets)
        # length_loss_l2 = self.mse_loss(epochlen_predictions, epochlen_targets)
        length_loss_ce = self.ce_loss(epochlen_predictions, epochlen_targets_bucket)

        #total_loss = (
        #    com_loss + duration_loss_l1 + duration_loss_l2 + length_loss_ce
        #)
        
        total_loss = (
            mel_loss_l1 + mel_loss_l2 + phase_loss_l1 + phase_loss_l2 + duration_loss_l1 + duration_loss_l2 + length_loss_ce
        )

        return (
            total_loss,
            mel_loss_l1,
            mel_loss_l2,
            phase_loss_l1,
            phase_loss_l2,
            #com_loss,
            duration_loss_l1,
            duration_loss_l2,
            length_loss_ce
        )
