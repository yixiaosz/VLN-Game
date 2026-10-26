#!/usr/bin/env python3

import argparse
import os
import random
import logging

from typing import Dict
import numpy as np
import torch
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio
import threading
from multiprocessing import Process, Queue

import sys
sys.path.append(".")
import time

from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import VLObjectNavEpisode
from agents.vlnav_agent_vlm import VLObjectNav_Agent
from habitat.utils.visualizations import maps
from utils.equ_trace import EquilibriumTraceRun
from utils.chat_utils import trace_configuration
from agents.system_prompt import Parsing_prompt, generator_prompt, discriminator_prompt
from utils.candidate_images import (
    CandidateImageRequests, CandidateRenderError, CandidateRenderCancelled,
    PROTOCOL as CANDIDATE_IMAGE_PROTOCOL, RENDER_TIMEOUT_SECONDS)


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# def generate_point_cloud(window):
def main(args, send_queue, receive_queue, candidate_images=None, candidate_renderer=None):
    args.exp_name = "vlobjectnav-"+ args.vln_mode# + "-mini"
    
    if args.gpt_type == 3:
        args.exp_name += "-mini"

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    args.task_config = "vlobjectnav_hm3d.yaml"
    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)

    config.defrost()
    # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    env = Env(config=config)

    follower = ShortestPathFollowerCompat(
        env._sim, 0.1, False
    )

    agent = VLObjectNav_Agent(args, follower, logging, candidate_images=candidate_images,
                             candidate_renderer=candidate_renderer)

    agg_metrics: Dict = defaultdict(float)

    num_episodes = len(env.episodes)
    if args.episode_count > -1:
        num_episodes = min(args.episode_count, len(env.episodes))
    print("num_episodes: ", num_episodes)

    trace_run = None
    if args.vln_mode == 'vlm_game':
        trace_run = EquilibriumTraceRun(
            log_dir, {'mode': args.vln_mode, 'requested_episodes': num_episodes,
                      'config_seed': config.SEED, 'argument_seed': args.seed,
                      'visualize': args.visualize, 'task_config': str(config),
                      'candidate_image_protocol': CANDIDATE_IMAGE_PROTOCOL,
                      'candidate_render_timeout_seconds': RENDER_TIMEOUT_SECONDS if candidate_renderer is None else None,
                      'candidate_renderer': candidate_renderer.configuration() if candidate_renderer is not None
                                            else {'backend': 'open3d_gui'},
                      'api_configuration': trace_configuration()},
            {'parsing': Parsing_prompt, 'generator': generator_prompt,
             'discriminator': discriminator_prompt}, logging)

    fail_case = {}
    fail_case['collision'] = 0
    fail_case['success'] = 0
    fail_case['detection'] = 0
    fail_case['exploration'] = 0

    count_episodes = 0
    start = time.time()
    
    while count_episodes < num_episodes:
        obs = env.reset()
        
        agent.reset()
        episode_trace = None
        if trace_run is not None:
            episode_trace = trace_run.start_episode(
                count_episodes + 1, env.current_episode.episode_id,
                env.current_episode.scene_id, obs['instruction']['text'])
            agent.episode_trace = episode_trace
        print("Instrcution: ", obs["instruction"]['text'])

        image = transform_rgb_bgr(obs["rgb"])  # 224*224*3
        image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
        # cv2.imshow("RGB0", obs["rgb"])

        
        logging.info(obs["instruction"]['text'])

        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:
            if args.gui_closed:
                break
            
            agent_state = env.sim.get_agent_state()
            try:
                action = agent.act(obs, agent_state, send_queue, receive_queue)

                if action == None:
                    continue
                obs = env.step(action)
            except CandidateRenderCancelled:
                args.gui_closed = True
                break
            except KeyboardInterrupt:
                if episode_trace is not None:
                    episode_trace.finish('interrupted', count_steps)
                if candidate_images is not None:
                    candidate_images.request_gui_close()
                raise
            except Exception as exc:
                if episode_trace is not None:
                    episode_trace.finish('error', count_steps,
                                         error={'type': type(exc).__name__, 'message': str(exc)})
                if candidate_images is not None:
                    candidate_images.request_gui_close()
                raise

            count_steps += 1
            
            # top_down_map = draw_top_down_map(env.get_metrics(), obs["rgb"].shape[0])
            # cv2.imshow("top_down_map", top_down_map)
            # cv2.waitKey(1)

        if args.gui_closed:
            if episode_trace is not None:
                episode_trace.finish('interrupted', count_steps, metrics=env.get_metrics())
            print("GUI closed; stopping run.")
            break

        if (
            action == 0 and 
            env.get_metrics()["spl"]
        ):
            # print("you successfully navigated to destination point")
            fail_case['success'] += 1
        else:
            # print("your navigation was not successful")
            if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                fail_case['exploration'] += 1
            elif agent.replan_count > 20:
                fail_case['collision'] += 1
            else:
                fail_case['detection'] += 1



        count_episodes += 1

        end = time.time()
        time_elapsed = time.gmtime(end - start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_steps),
            "FPS {},".format(int(count_steps / (end - start_ep)))
        ]) + '\n'

        log += "Failed Case: collision/exploration/detection/success/total:"
        log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
            np.sum(fail_case['collision']),
            np.sum(fail_case['exploration']),
            np.sum(fail_case['detection']),
            np.sum(fail_case['success']),
            count_episodes) + '\n'
        
        metrics = env.get_metrics()
        if metrics['distance_to_goal'] != np.inf:
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
                        

        log += "Metrics: "
        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes) + '\n'
        
        log += ", ".join(m + ": {:.3f}".format(v) for m,v in metrics.items())
        print(log)
        logging.info(log)
        if episode_trace is not None:
            episode_trace.finish('completed', count_steps, metrics=metrics)


        

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    for stat_key in avg_metrics.keys():
        logger.info("{}: {:.3f}".format(stat_key, avg_metrics[stat_key]))
        print("{}: {:.3f}".format(stat_key, avg_metrics[stat_key]))
    return


def visualization_thread(send_queue, receive_queue, candidate_images=None):
    import open3d.visualization.gui as gui
    from utils.vis_gui import ReconstructionWindow

    app_win = None
    try:
        app = gui.Application.instance
        app.initialize()
        mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
        app_win = ReconstructionWindow(args, mono, send_queue, receive_queue, candidate_images)
        if candidate_images is not None:
            candidate_images.set_gui_close_callback(
                lambda: app.post_to_main_thread(app_win.window, app_win.window.close))
        app.run()
    finally:
        args.gui_closed = True
        if app_win is not None:
            app_win.is_done = True
        if candidate_images is not None:
            candidate_images.close()


if __name__ == "__main__":
    args = get_args()
    args.gui_closed = False

    send_queue = Queue()
    receive_queue = Queue()
    candidate_images = CandidateImageRequests() if args.visualize and 'vlm' in args.vln_mode else None
    candidate_renderer = None

    gui_started = bool(args.visualize)
    if gui_started:
        # Create a thread for the Open3D visualization
        visualization = threading.Thread(target=visualization_thread, args=(send_queue, receive_queue, candidate_images))
        visualization.start()

    # Run ROS code in the main thread
    exit_code = 0
    try:
        if not gui_started and 'vlm' in args.vln_mode:
            from utils.candidate_renderer import CandidateRenderer
            candidate_renderer = CandidateRenderer()
        main(args, send_queue, receive_queue, candidate_images, candidate_renderer)
    except KeyboardInterrupt:
        if candidate_images is not None:
            candidate_images.request_gui_close()
        exit_code = 130
    except Exception:
        logging.exception('Test stopped on error; incomplete episode was not counted')
        if candidate_images is not None:
            candidate_images.request_gui_close()
        exit_code = 1
    finally:
        if candidate_renderer is not None:
            candidate_renderer.close()

    if gui_started:
        visualization.join()
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(exit_code)
    sys.exit(exit_code)
