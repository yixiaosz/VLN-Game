import numpy as np
import torch
import torch.nn.functional as F
import ast
import concurrent.futures
from collections import Counter
from utils.chat_utils import chat_with_gpt, chat_with_gpt4v
from agents.system_prompt import (
    Parsing_prompt,
    discriminator_prompt,
    generator_prompt
    )


class Equilibrium_Ranking():
    r""" EQUILIBRIUM - RANKING : LM R ANKING VIA EQUILIBRIUM SEARCH
    1. generate initial policies (only focus on the signalling policies)
    2. update policies using No-regret learning algorithms
    3. make decision based on the learned policies
    
    Args:
    """
    def __init__(self, text_queries, logger=None, episode_trace=None):
        
        # Hyperparameters
        self.lambda_G = 0.1
        self.lambda_D = 0.1
        self.eta_G = 0.1
        self.eta_D = 0.1
        self.iterations = 500
        self.bise = 0.05
        self.trace = None
        
        # target_hint = "Hint: the target category is " + object_category
        
        self.text_description = text_queries #+ target_hint
        message = []
        message.append({"role": "system", "content": Parsing_prompt})
        message.append({"role": "user", "content": self.text_description})
        self.response_message = chat_with_gpt(message)
        if episode_trace is not None:
            episode_trace.emit('parsing_response', raw_content=self.response_message,
                               prompt_ref='run.json#/prompts/parsing')
        
        if logger != None:
            self.logger = logger
        
    def equilibrium_search(self, base64_image_list, candidate_id, trace=None):
        self.trace = trace
        try:
            if trace is not None:
                trace.iterations = self.iterations
                trace.emit('search_parameters', iterations=self.iterations,
                           generator_samples=5, discriminator_samples=5, bias=self.bise,
                           lambda_G=self.lambda_G, lambda_D=self.lambda_D,
                           eta_G=self.eta_G, eta_D=self.eta_D)
            result = self._equilibrium_search(base64_image_list, candidate_id)
            if trace is not None:
                trace.finish('completed' if trace.initialized else 'early_reject',
                             selected_candidate_id=result,
                             selected_policy_index=self.selected_policy_index if trace.initialized else None,
                             reason=None if trace.initialized else 'generator_support_contains_only_reject_option')
            return result
        except Exception as exc:
            if trace is not None:
                trace.finish('error', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            self.trace = None

    def _equilibrium_search(self, base64_image_list, candidate_id):

        self.candidate_id = candidate_id
        chat_history_for_generator_correct = []
        chat_history_for_discriminator = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": generator_prompt})
        chat_history_for_discriminator.append({"role": "system", "content": discriminator_prompt})
        
        # generate initial policies 
        # Get generative probabilities for "correct"
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": self.text_description,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        chat_history_for_generator_correct.append({"role": "user", "content": image_contents})
        generative_idx, generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct)
        
        generative_probabilities_incorrect = [1 - value for value in generative_probabilities_correct]
        
        if len(generative_idx) == 1:
            return -1
        
        
        discriminative_probabilities = self._get_discriminative_probabilities(chat_history_for_discriminator, base64_image_list, generative_idx)
        
        plm_y_given_x_v = {
            'correct': [x + self.bise for x in generative_probabilities_correct],
            'incorrect': [x + self.bise for x in generative_probabilities_incorrect]
        }
        plm_v_given_x_y = discriminative_probabilities
        
        
        pi_G_1 = self._normalize_initial_policy(plm_y_given_x_v, axis=0)
        pi_D_1 = self._normalize_initial_policy(plm_v_given_x_y, axis=0)
        pi_G = pi_G_1
        pi_D = pi_D_1
        if self.trace is not None:
            self.trace.emit('policy_layout', policy_candidate_order=generative_idx,
                            verdict_order=['correct', 'incorrect'],
                            discriminator_key_to_candidate_id=dict(zip(plm_v_given_x_y, generative_idx)),
                            generator_normalization_input=plm_y_given_x_v,
                            discriminator_normalization_input=plm_v_given_x_y)
            self.trace.snapshot(0, pi_G, pi_D)
        # Print initial policies
        print("Initial Generator Policy:", pi_G_1)
        print("Initial Discriminator Policy:", pi_D_1)
        
        v_values = ['correct', 'incorrect']
        y_values = list(plm_v_given_x_y.keys())
        
        # Initialize Q values
        Q_G = {v: np.zeros(len(y_values)) for v in v_values}
        Q_D = {y: np.zeros(len(v_values)) for y in y_values}
        
        # Equilibrium ranking
        for t in range(1, self.iterations + 1):
            # Update Q values
            for v in v_values:
                for i, y in enumerate(y_values):
                    Q_G[v][i] += (1 / (2 * t)) * pi_D[y][v_values.index(v)]

            for y in y_values:
                for i, v in enumerate(v_values):
                    Q_D[y][i] += (1 / (2 * t)) * pi_G[v][y_values.index(y)]

            # Update policies
            pi_G = self._update_policy(Q_G, pi_G_1, self.lambda_G, self.eta_G, t)
            pi_D = self._update_policy(Q_D, pi_D_1, self.lambda_D, self.eta_D, t)
            if self.trace is not None:
                self.trace.updates_completed = t
                if t in {self.iterations // 4, self.iterations // 2,
                         3 * self.iterations // 4, self.iterations}:
                    self.trace.snapshot(t, pi_G, pi_D)

        # Display the final policies
        print("Final Generator Policy:", pi_G)
        print("Final Discriminator Policy:", pi_D)
        
        self.selected_policy_index = pi_G['correct'].argmax(axis=0)
        return generative_idx[self.selected_policy_index]
        # else:
        #     return generative_probabilities_correct.index(max(generative_probabilities_correct))
                        
    def generator_search(self, base64_image_list, candidate_id, num_samples=1):

        self.candidate_id = candidate_id
        
        chat_history_for_generator_correct = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": generator_prompt})
        
        # generate initial policies 
        # Get generative probabilities for "correct"
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": self.text_description,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        chat_history_for_generator_correct.append({"role": "user", "content": image_contents})
        generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct, num_samples=num_samples)
        
        max_index = generative_probabilities_correct.index(max(generative_probabilities_correct))
        return max_index

    # Normalizing initial policies
    def _normalize_initial_policy(self, plm_dict, axis=0):
        norm_dict = {}
        for key in plm_dict:
            norm_dict[key] = plm_dict[key] / np.sum(plm_dict[key], axis=axis)
        return norm_dict
    
    # Function to update policies
    def _update_policy(self, Q, pi, lambd, eta, t):
        updated_pi = {}
        for key in Q:
            exponent = (Q[key] + lambd * np.log(pi[key])) / (1 / (eta * t) + lambd)
            updated_pi[key] = np.exp(exponent) / np.sum(np.exp(exponent))
        return updated_pi

    def _get_generative_probabilities(self, chat_history, num_samples=5):
        id_list = []
        responses = []
        response_ids = []
        if self.trace is not None:
            self.trace.emit('query_batch', role='generator', requested_samples=num_samples,
                            input_positions=list(range(len(self.candidate_id))))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_message = {executor.submit(chat_with_gpt4v, chat_history): (chat_history, f'g-{n + 1}') for n in range(num_samples)}
            for completion_order, future in enumerate(concurrent.futures.as_completed(future_to_message), 1):
                prompt, sample_id = future_to_message[future]
                try:
                    response = future.result()
                    responses.append(response)
                    response_ids.append(sample_id)
                except Exception as exc:
                    if self.trace is not None:
                        self.trace.emit('vlm_response', role='generator', sample_id=sample_id,
                                        completion_order=completion_order, status='error',
                                        error={'type': type(exc).__name__, 'message': str(exc)})
                    print(f'{prompt} generated an exception: {exc}')
                else:
                    if self.trace is not None:
                        self.trace.emit('vlm_response', role='generator', sample_id=sample_id,
                                        completion_order=completion_order, raw_content=response,
                                        status='returned')

        for i, response in enumerate(responses):
            if self.trace is None:
                self.logger.info(response)
            try:
                ground_json = ast.literal_eval(response)
                id_list.append(ground_json["object_id"])
            except Exception as exc:
                if self.trace is not None:
                    self.trace.emit('sample_parse_error', role='generator', sample_id=response_ids[i],
                                    error={'type': type(exc).__name__, 'message': str(exc)})
                raise
            if self.trace is not None:
                self.trace.emit('sample_parsed', role='generator', sample_id=response_ids[i],
                                parsed_object_id=ground_json['object_id'])
            
        counts = Counter(id_list)
        total_count = len(id_list)
        # Calculate the probabilities
        probabilities = {number: count / total_count for number, count in counts.items()}

        probabilities_idx = []
        probabilities_value = []
        for i in (self.candidate_id ):
            if str(i) in probabilities:
                probabilities_idx.append(i)
                probabilities_value.append(probabilities[str(i)])
        print('probabilities_idx: ', probabilities_idx)
        probabilities_idx.append(-1)
        if '-1' in probabilities:
            probabilities_value.append(probabilities['-1'])
        else:
            probabilities_value.append(0)
        if self.trace is not None:
            self.trace.emit('generator_aggregate', sample_ids=response_ids,
                            requested_samples=num_samples, denominator_used=total_count,
                            counts=counts, p_vlm_observed=probabilities,
                            p_vlm_all_candidates={str(i): probabilities.get(str(i), 0)
                                                  for i in list(self.candidate_id) + [-1]},
                            policy_candidate_order=probabilities_idx,
                            p_vlm_correct=probabilities_value,
                            p_vlm_incorrect=[1 - value for value in probabilities_value])
        return probabilities_idx, probabilities_value
        
    def _get_discriminative_probabilities(self, chat_history, base64_image_list, generative_idx, num_samples=5):
        probabilities = {}
        # Preserve the reference accumulation bug until separately approved.
        responses = []
        response_ids = []
        candidate_batches = []
        candidate_mapping = {}
        gene_idx = 0
        for i, idx in enumerate(self.candidate_id):
            if idx in generative_idx:
                new_sample_ids = []
                input_position = i
                if self.trace is not None:
                    self.trace.emit('query_batch', role='discriminator', candidate_id=idx,
                                    requested_samples=num_samples, input_positions=[i])
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_prompt = {executor.submit(chat_with_gpt4v, chat_history + self.create_payload(self.text_description, [base64_image_list[i]]), json_mode=False): (base64_image_list[i], f'd-{idx}-{n + 1}') for n in range(num_samples)}
                    for completion_order, future in enumerate(concurrent.futures.as_completed(future_to_prompt), 1):
                        prompt, sample_id = future_to_prompt[future]
                        try:
                            response = future.result()
                            responses.append(response)
                            response_ids.append(sample_id)
                            new_sample_ids.append(sample_id)
                        except Exception as exc:
                            if self.trace is not None:
                                self.trace.emit('vlm_response', role='discriminator', candidate_id=idx,
                                                sample_id=sample_id, completion_order=completion_order,
                                                status='error', error={'type': type(exc).__name__, 'message': str(exc)})
                            print(f'{prompt} generated an exception: {exc}')
                        else:
                            if self.trace is not None:
                                self.trace.emit('vlm_response', role='discriminator', candidate_id=idx,
                                                sample_id=sample_id, completion_order=completion_order,
                                                input_positions=[input_position], raw_content=response,
                                                status='returned')
                            
                probabilities[gene_idx] = 0
                for i, response in enumerate(responses):
                    if self.trace is None:
                        self.logger.info(response)
                    discriminate = response.strip().lower().strip(".\"'`")
                    if self.trace is not None and response_ids[i] in new_sample_ids:
                        self.trace.emit('sample_parsed', role='discriminator', candidate_id=idx,
                                        sample_id=response_ids[i], parsed_verdict=discriminate,
                                        recognized_verdict=discriminate in ('correct', 'incorrect'))
                    if discriminate == "correct":
                        probabilities[gene_idx] +=1
                    else:
                        probabilities[gene_idx] += 0
                        
                candidate_mapping[gene_idx] = idx
                if self.trace is not None:
                    batch = dict(candidate_id=idx, internal_key=gene_idx,
                                 new_sample_ids=new_sample_ids,
                                 sample_pool_ids_used_by_current_code=list(response_ids),
                                 correct_count_used=probabilities[gene_idx], denominator_used=num_samples)
                    candidate_batches.append(batch)
                    self.trace.emit('discriminator_candidate_score', **batch)
                gene_idx += 1

        if gene_idx == 1:
            probabilities[-1] = num_samples - probabilities[0]
            reject_all = dict(source='derived_complement', count=probabilities[-1],
                              denominator_used=num_samples, additional_logical_samples=0,
                              source_internal_key=0)
        else:
            responses = []
            reject_sample_ids = []
            if self.trace is not None:
                self.trace.emit('query_batch', role='discriminator', candidate_id=-1,
                                scope='all_input_images', requested_samples=num_samples,
                                input_positions=list(range(len(base64_image_list))))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_prompt = {executor.submit(chat_with_gpt4v, chat_history + self.create_payload(self.text_description, base64_image_list), json_mode=False): (base64_image_list, f'd-reject-{n + 1}') for n in range(num_samples)}
                for completion_order, future in enumerate(concurrent.futures.as_completed(future_to_prompt), 1):
                    prompt, sample_id = future_to_prompt[future]
                    try:
                        response = future.result()
                        responses.append(response)
                        reject_sample_ids.append(sample_id)
                    except Exception as exc:
                        if self.trace is not None:
                            self.trace.emit('vlm_response', role='discriminator', candidate_id=-1,
                                            sample_id=sample_id, completion_order=completion_order,
                                            status='error', error={'type': type(exc).__name__, 'message': str(exc)})
                        print(f'{prompt} generated an exception: {exc}')
                    else:
                        if self.trace is not None:
                            self.trace.emit('vlm_response', role='discriminator', candidate_id=-1,
                                            scope='all_input_images', sample_id=sample_id,
                                            completion_order=completion_order, raw_content=response,
                                            status='returned')
            probabilities[-1] = 0            
            for i, response in enumerate(responses):
                discriminate = response.strip().lower().strip(".\"'`")
                if self.trace is not None:
                    self.trace.emit('sample_parsed', role='discriminator', candidate_id=-1,
                                    sample_id=reject_sample_ids[i], parsed_verdict=discriminate,
                                    recognized_verdict=discriminate in ('correct', 'incorrect'))
                if discriminate == "incorrect":
                    probabilities[-1] +=1
                else:
                    probabilities[-1] += 0

                
            reject_all = dict(source='queried_all_input_images', count=probabilities[-1],
                              sample_ids=reject_sample_ids, denominator_used=num_samples)

        if self.trace is not None:
            candidate_mapping[-1] = -1
            self.trace.emit('discriminator_aggregate', candidate_batches=candidate_batches,
                            internal_key_to_candidate_id=candidate_mapping,
                            verdict_order=['correct', 'incorrect'], reject_all=reject_all,
                            counts_used=probabilities, denominator_used=num_samples,
                            p_vlm_before_bias={key: [value/num_samples, 1-value/num_samples]
                                               for key, value in probabilities.items()},
                            bias=self.bise,
                            normalization_input={key: [value/num_samples + self.bise,
                                                       1-(value/num_samples) + self.bise]
                                                 for key, value in probabilities.items()},
                            values_outside_probability_range=any(
                                value < 0 or value > num_samples for value in probabilities.values()))
        print(probabilities)            
        return {key:[value/num_samples + self.bise, 1-(value/num_samples) + self.bise] for key, value in probabilities.items()}
    
    # def _get_discriminative_probabilities(self, chat_history, base64_image_list, num_samples=5):
    #     probabilities = {}
    #     for _ in range(num_samples):
    #         responses = []
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 future_to_prompt = {executor.submit(chat_with_gpt4v, self.create_payload(chat_history, self.text_description, base64_image)): base64_image for base64_image in base64_image_list}
    #                 for future in concurrent.futures.as_completed(future_to_prompt):
    #                     prompt = future_to_prompt[future]
    #                     try:
    #                         response = future.result()
    #                         responses.append(response)
    #                     except Exception as exc:
    #                         print(f'{prompt} generated an exception: {exc}')
                            
    #         incorrect_num = 0
    #         for i, response in enumerate(responses):
    #             # self.logger.info(response)
    #             ground_json = ast.literal_eval(response)
    #             discriminate = ground_json["correctness"] 
    #             if discriminate == "correct":
    #                 if i not in probabilities:
    #                     probabilities[i] = 1 
    #                 else:
    #                     probabilities[i] += 1
    #             else:
    #                 incorrect_num += 1
    #                 if i not in probabilities:
    #                     probabilities[i] = 0
    #                 else:
    #                     probabilities[i] += 0
                        
    #         if incorrect_num == len(responses): # reject all the objects
    #             if -1 not in probabilities:
    #                 probabilities[-1] = 1
    #             else:
    #                 probabilities[-1] += 1
    #         else:
    #             probabilities[-1] = 0
                
    #     print(probabilities)            
    #     return {key:[value/num_samples, 1-(value/num_samples)] for key, value in probabilities.items()}
    
    def discriminate_text(self, id):
        return "The candidate object {:d} match the user's description. Is it correct?".format(id) + '\n'



    def create_payload(self, text, base64_image_list):
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": text,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
            })
        
        return [{"role": "user", "content": image_contents}]
