"""Batched causal generation must preserve per-request prompts and EOS bounds."""
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
from polytopia_bot.learning.serving import BatchGenerator


class Tokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs):return messages[0]['tokens']
    def decode(self,tokens,**kwargs):return ','.join(map(str,tokens))


class Model:
    device='cpu'
    def generate(self,**kwargs):
        assert kwargs['input_ids'].tolist()==[[0,0,11],[11,12,13]]
        assert kwargs['attention_mask'].tolist()==[[0,0,1],[1,1,1]]
        return torch.cat([kwargs['input_ids'],torch.tensor([[8,2,0],[9,10,2]])],dim=1)


def test_batch_inputs_left_pad_and_do_not_log_padding_as_response():
    generator=object.__new__(BatchGenerator)
    generator.tokenizer=Tokenizer();generator.model=Model()
    generator.config=dict(max_length=10,max_new_tokens=3)
    result=generator.batch([[{'tokens':[11]}],[{'tokens':[11,12,13]}]])
    assert result[0].prompt_token_ids==[11]
    assert result[0].response_token_ids==[8,2]
    assert result[1].response_token_ids==[9,10,2]
    assert result[0].text=='8' and result[1].text=='9,10'
    assert generator.batch([])==[]
    with pytest.raises(ValueError,match='context budget'):
        generator.batch([[{'tokens':list(range(9))}]])


def test_tiny_qwen_batch_matches_single_greedy_decisions():
    from transformers import Qwen3Config,Qwen3ForCausalLM,set_seed
    from polytopia_bot.learning.smoke import tiny_tokenizer
    set_seed(5)
    torch.set_num_threads(2)
    generator=object.__new__(BatchGenerator)
    generator.tokenizer=tiny_tokenizer()
    generator.config=dict(max_length=256,max_new_tokens=6)
    generator.model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(generator.tokenizer),hidden_size=32,
        intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=256,tie_word_embeddings=True)).eval()
    requests=[[dict(role='user',content=text)] for text in ('choose one action','choose one action with a much longer public catalog')]
    singles=[generator(request) for request in requests]
    batched=generator.batch(requests)
    assert [g.prompt_token_ids for g in singles]==[g.prompt_token_ids for g in batched]
    assert [g.response_token_ids for g in singles]==[g.response_token_ids for g in batched]
