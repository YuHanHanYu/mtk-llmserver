#ifndef LLMSERVER_H
#define LLMSERVER_H

#ifdef __cplusplus
extern "C" {
#endif

void llmserver_start(const char* json_params);
void llmserver_stop();

#ifdef __cplusplus
}
#endif

#endif /* LLMSERVER_H */
