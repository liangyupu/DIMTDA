import os
import torch
from transformers import PreTrainedModel, Blip2QFormerConfig, Blip2QFormerModel, GenerationConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

class DIMTDAModel(PreTrainedModel):
    def __init__(self, config, trans_model, dit_model, nougat_model, num_query_tokens, qformer_config_dir):
        super().__init__(config)
        self.dit_encoder = dit_model
        self.nougat_encoder = nougat_model.encoder
        
        self.dit_enc_to_dec_proj = nn.Linear(1024, 512)
        self.nougat_enc_to_dec_proj = nn.Linear(1024, 512)
        self.trans_enc_to_dec_proj = nn.Linear(1024, 512)
        
        self.trans_decoder = trans_model.decoder

        self.expand_feature_net = nn.Linear(768, 1024)
        
        qformer_config = Blip2QFormerConfig.from_pretrained(qformer_config_dir)
        qformer_config.num_attention_heads = 8
        qformer_config.num_hidden_layers = 1
        qformer_config.encoder_hidden_size = 1024
        qformer_config.intermediate_size = 2048
        qformer_config.hidden_size = 1024
        self.nougat_qformer = Blip2QFormerModel(config=qformer_config)
        self.dit_qformer = Blip2QFormerModel(config=qformer_config)
        
        self.num_query_tokens = num_query_tokens
        
        self.nougat_query_tokens = nn.Parameter(torch.zeros(1, self.num_query_tokens, qformer_config.hidden_size))
        self.dit_query_tokens = nn.Parameter(torch.zeros(1, self.num_query_tokens, qformer_config.hidden_size))
        
        self.W_alpha = nn.Linear(2048, 1)
        
    def forward(
        self,
        dit_pixel_values: Optional[torch.FloatTensor] = None,
        nougat_pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=True,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        nougat_encoder_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        nougat_encoder_hidden_states = nougat_encoder_outputs.last_hidden_state
        
        dit_encoder_outputs = self.dit_encoder(
            pixel_values=dit_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        dit_encoder_hidden_states = dit_encoder_outputs.last_hidden_state
        expanded_dit_encoder_hidden_states = self.expand_feature_net(dit_encoder_hidden_states)
        
        # qformer
        nougat_query_tokens = self.nougat_query_tokens.expand(nougat_encoder_hidden_states.shape[0], -1, -1)
        dit_query_tokens = self.dit_query_tokens.expand(expanded_dit_encoder_hidden_states.shape[0], -1, -1)
        
        nougat_query_outputs = self.nougat_qformer(
            query_embeds=nougat_query_tokens,
            encoder_hidden_states=nougat_encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        dit_query_outputs = self.dit_qformer(
            query_embeds=dit_query_tokens,
            encoder_hidden_states=expanded_dit_encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        alpha = torch.sigmoid(self.W_alpha(torch.cat([nougat_query_outputs[0], dit_query_outputs[0]], dim=-1)))
        query_output = alpha * nougat_query_outputs[0] + (1 - alpha) * dit_query_outputs[0]
        
        encoder_hidden_states = self.trans_enc_to_dec_proj(query_output)
        
        # Decode
        decoder_outputs = self.trans_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct_trans = CrossEntropyLoss()
            loss_trans = loss_fct_trans(logits.reshape(-1, self.trans_decoder.config.vocab_size), labels.reshape(-1).long())
            
            loss = loss_trans

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )
    
    def generate(
        self,
        dit_pixel_values: Optional[torch.FloatTensor] = None,
        nougat_pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=True,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        
        nougat_encoder_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        nougat_encoder_hidden_states = nougat_encoder_outputs.last_hidden_state
        
        dit_encoder_outputs = self.dit_encoder(
            pixel_values=dit_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        dit_encoder_hidden_states = dit_encoder_outputs.last_hidden_state
        expanded_dit_encoder_hidden_states = self.expand_feature_net(dit_encoder_hidden_states)
        
        # qformer
        nougat_query_tokens = self.nougat_query_tokens.expand(nougat_encoder_hidden_states.shape[0], -1, -1)
        dit_query_tokens = self.dit_query_tokens.expand(expanded_dit_encoder_hidden_states.shape[0], -1, -1)
        
        nougat_query_outputs = self.nougat_qformer(
            query_embeds=nougat_query_tokens,
            encoder_hidden_states=nougat_encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        dit_query_outputs = self.dit_qformer(
            query_embeds=dit_query_tokens,
            encoder_hidden_states=expanded_dit_encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        alpha = torch.sigmoid(self.W_alpha(torch.cat([nougat_query_outputs[0], dit_query_outputs[0]], dim=-1)))
        query_output = alpha * nougat_query_outputs[0] + (1 - alpha) * dit_query_outputs[0]
        
        encoder_hidden_states = self.trans_enc_to_dec_proj(query_output)
        
        generation_outputs = self.trans_decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            generation_config=generation_config,
        )
        
        return generation_outputs
